import pyopencl as cl
import numpy as np
# import matplotlib.pyplot as plt

kernel_image = '''
float2 complexconj(float2 a) {
    return (float2) {a.x, -a.y};
}


float2 complexmul(float2 a, float2 b) {
    return (float2) { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
}


float4 matmulvec(float2* mat, float4 vec) {
    float4 out;
    out.xy = complexmul(mat[0], vec.xy) + complexmul(mat[1], vec.zw);
    out.zw = complexmul(mat[2], vec.xy) + complexmul(mat[3], vec.zw);
    return out;
}


__kernel void amplitude2intensity(__global float4 *in_out) {
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    int ysiz = get_global_size(1);
    int id = xid * ysiz + yid;

    float2 f_0 = in_out[id].xy;
    float2 f_1 = in_out[id].zw;

    float2 bf = complexmul(f_0, complexconj(f_0));
    float2 df = complexmul(f_1, complexconj(f_1));

    in_out[id] = (float4) {bf, df};
}


__kernel void propagate_wave(__global float4 *in_out, float t,
                             __global float *s_array, __global float *Xg_array,
                             float X0_im) {

    int xid = get_global_id(0);
    int xsiz = get_global_size(0);
    int yid = get_global_id(1);
    int ysiz = get_global_size(1);
    // position of the current pixel in the linear array
    int id = xid * ysiz + yid;

    //stacking fault phase shift, in case it is needed for some other example
    float alpha = 0.0f;

    //s for this pixel
    float s = s_array[id];

    // Real and imaginary extinction distances
    float Xg_re = Xg_array[id];
    float Xg_im = X0_im * 1.16;

    // variables to make the equations neate
    float f_1 = sqrt(s*s + 1.0f / (Xg_re*Xg_re));
    float f_2 = Xg_re * f_1;

    //NB gamma and q are twice the values given in HHMPW
    //because the 1/2 cancels *XX*
    float2 gamma = (float2) { (s - f_1), (s + f_1) };

    //anomalous absorption parameter q
    float2 q = (float2) { 1.0f/X0_im - 1.0f/(Xg_im*f_2),
                         1.0f/X0_im + 1.0f/(Xg_im*f_2) };

    //beta/2 determines the strength of coupling between the BF & DF beams
    float beta_2 = 0.5f * atan( 1.0f / (s * Xg_re ) );

    //use complex notation for all matrix elements {Re, Im}
    float2 sin_beta_2 = (float2) {sin(beta_2), 0.0f};
    float2 cos_beta_2 = (float2) {cos(beta_2), 0.0f};

    //write complex exponentials as {cos, sin} using de Moivre
    float2 exp_alpha = (float2) {cos(alpha), sin(alpha)};
    //helpful to define complex conjugate too
    float2 exp_alphaN = (float2) {cos(alpha), -sin(alpha)};

    //scattering matrix formulation, all numbers are complex
    float2 big_c[4] = {cos_beta_2, sin_beta_2, -sin_beta_2.x * exp_alpha,
                       cos_beta_2.x * exp_alpha};
    float2 big_c_inv[4] = {cos_beta_2, -sin_beta_2.x * exp_alphaN, sin_beta_2,
                           cos_beta_2.x * exp_alphaN};

    //NB note exp(pi) not exp(2pi) because we have cancelled the factor 2 *XX*
    float2 big_g_g = M_PI_F * gamma * t;
    float2 big_g_q = -M_PI_F * q * t;

    //we use exp(i(A+iB))=exp(-B)*(cosA+i.sin(A))
    float2 big_g_0 = exp(big_g_q.x) * (float2) { cos(big_g_g.x),
                                                sin(big_g_g.x) };
    float2 big_g_3 = exp(big_g_q.y) * (float2) { cos(big_g_g.y),
                                                sin(big_g_g.y) };

    float2 big_g[4] = {big_g_0, 0.0f, 0.0f, big_g_3};

    float4 out = matmulvec(big_c, matmulvec(big_g, matmulvec(big_c_inv,
                                                             in_out[id])));
//    changed output for debugging
//    float4 out = {s,0.0f,z, 0.0f};
    in_out[id] = out;
}

__kernel void calculate_dR(__global float *dR_x, __global float *dR_y,
                            __global float *dR_z, float pix2nm, float z_,
                            float c, float alpha, float n,
                            float pos, float sense, float u) {

    int xid = get_global_id(0);
    int yid = get_global_id(1);
    int xsiz = get_global_size(0);
    int ysiz = get_global_size(1);
    int id = xid * ysiz + yid;

    // helical and cycloidal amplitudes
    float helic_amp = 0.2f;
    float cyclic_amp = 0.2f;

    float x_= u*(( (float) xid - pos ) * pix2nm);//NB=0 when u=0
    float y_= (1.0f-u)*( ((float) yid - pos ) * pix2nm );//NB=0 when u=1
    float r = sqrt(x_* x_+ y_* y_+ z_* z_);
    // Ti displacement using an orbital model, dR=c*r^n*exp(-alpha*r^n)
    // constant d to make the calcs neater
    float d = sense * c * pow(r,n) * exp(-alpha * pow(r,n));
    // version with varying strength of c along vortex
    // float d = sense * c * pow(r,n) * exp(-alpha * pow(r,n))
    //   * (1.0f + helic_amp * sin(24.0f * M_PI_F*(float) xid / (float) xsiz));
    // dR = d*[-z/r,0,y/r] for y-array
    // dR = d*[0,-z/r,x/r] for x-array
    dR_x[id] = - (1.0f-u) * z_* d / r;//NB=0 when u=1
    dR_y[id] = - u * z_* d / r;//NB=0 when u=0}
    // version with cycloidal displacements for the first set of vortices
    // dR_y[id] = - u * z_* d / r
    //           + cyclic_amp * sin(12.0f * M_PI_F*(float) xid / (float) xsiz);
    dR_z[id] = ((1.0f-u) * y_ + u * x_) * d / r;
}

__kernel void calculate_Xg(__global float *Xg, __global float *dR_x,
                           __global float *dR_y, __global float *dR_z,
                           float g_x, float g_y, float g_z, float a, float k) {

    //NB we need gmag in A^-1, not nm^-1, for Kirkland
    float gmag2 = 0.01f * (g_x*g_x + g_y*g_y + g_z*g_z) / (a*a);
    //Kirkland scattering factors for this gmag
    //O
    float f_O = 3.39969204E-01f / (gmag2 + 3.81570280E-01f) +
                3.07570172E-01f / (gmag2 + 3.81571436E-01f) +
                1.30369072E-01f / (gmag2 + 1.91919745E+01f) +
                8.83326058E-02f * exp(-gmag2 * 7.60635525E-01f) +
                1.96586700E-01f * exp(-gmag2 * 2.07401094E+00f) +
                9.96220028E-04f * exp(-gmag2 * 3.032668690E-02f);
    //Ti
    float f_Ti = 3.62383267E-01f / (gmag2 + 7.54707114E-02f) +
                 9.84232966E-01f / (gmag2 + 4.97757309E-01f) +
                 7.41715642E-01f / (gmag2 + 8.17659391E+00f) +
                 3.62555269E-01f * exp(-gmag2 * 9.55524906E-01f) +
                 1.49159390E+00f * exp(-gmag2 * 1.62221677E+01f) +
                 1.61659509E-02f * exp(-gmag2 * 7.33140839E-02f);
    //Pb
    float f_Pb = 1.00795975E+00f / (gmag2 + 1.17268427E-01f) +
                 3.09796153E+00f / (gmag2 + 8.80453235E-01f) +
                 3.61296864E+00f / (gmag2 + 1.47325812E+01f) +
                 2.62401476E-01f * exp(-gmag2 *1.43491014E-01f) +
                 4.05621995E-01f * exp(-gmag2 * 1.04103506E+00f) +
                 1.31812509E-02f * exp(-gmag2 * 2.39575415E-02f);

    //polarisation displacement
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    int xsiz = get_global_size(0);
    int ysiz = get_global_size(1);
    int id = xid * ysiz + yid;

    //Structure factor F - using deMoivre for real and complex parts
    //assuming Ti displacement only
    float F_re = f_Pb +
                 f_O * ( cos(M_PI_F*(g_x+g_y)) + cos(M_PI_F*(g_y+g_z))
                        + cos(M_PI_F*(g_z+g_x)) ) +
                 f_Ti* ( cos(M_PI_F*( g_x*(1.0f+2.0f*dR_x[id]) +
                                     g_y*(1.0f+2.0f*dR_y[id]) +
                                     g_z*(1.0f+2.0f*dR_z[id]))) );
    float F_im = f_O * (sin(M_PI_F*(g_x+g_y)) + sin(M_PI_F*(g_y+g_z)) +
                        sin(M_PI_F*(g_z+g_x))) +
                 f_Ti* ( sin(M_PI_F*(g_x*(1.0f+2.0f*dR_x[id]) +
                                     g_y*(1.0f+2.0f*dR_y[id]) +
                                     g_z*(1.0f+2.0f*dR_z[id]))) );
    float F = sqrt(F_re*F_re + F_im*F_im);

    //Bragg angle
    float theta = asin(sqrt(g_x*g_x + g_y*g_y + g_z*g_z) / (2.0f*a*k));

    //Extinction distance
    Xg[id] = M_PI_F * a*a*a*10 * k * cos(theta) / F;
}

__kernel void update(__global float *out_array, __global float *in1_array,
                     __global float *in2_array) {
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    int ysiz = get_global_size(1);
    int id = xid * ysiz + yid;

    out_array[id] = in1_array[id] + in2_array[id];
}


__kernel void calculate_s(__global float *s_array, __global float *dR_x,
                          __global float *dR_y, __global float *dR_z,
                          __global float *dRz_x, __global float *dRz_y,
                          __global float *dRz_z, float g_x, float g_y,
                          float g_z, float a, float dz, float s) {

    int xid = get_global_id(0);
    int yid = get_global_id(1);
    int xsiz = get_global_size(0);
    int ysiz = get_global_size(1);
    int id = xid * ysiz + yid;

    // g.R
    float gdotR = ( g_x * dR_x[id] + g_y * dR_y[id] + g_z * dR_z[id] );
    //& a little bit further along z
    float gdotR_dz = ( g_x * dRz_x[id] + g_y * dRz_y[id] + g_z * dRz_z[id] );
    //local value of deviation parameter is s + d(g.R)/dz
    //NB divide by a to get g in nm^-1
    // version with varying s along vortex
    // s_array[id] = s + 0.06f * ((float) xid / (float) xsiz)
    //              + (gdotR_dz - gdotR) / (a * dz);
    s_array[id] = s + (gdotR_dz - gdotR) / (a * dz);
}

'''


class ClHowieWhelan:

    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.image_prog = cl.Program(self.ctx, kernel_image).build()

    def calc_2D(self, xsiz, ysiz, pix2nm, z, k, s, X0i, dt, a, g,
                c0, c1, alpha, n, m):
        '''
         inputs:
         x-dimension(pix), y-coords(nm), z-coords(nm),
         wave vector(nm^-1), deviation parameter(nm^-1),
         zero-order extinction(nm),
         slice thickness(nm), lattice parameter(nm), g-vector magnitude(nm^-1),
         vortex displacement magnitude, polarisation magnitude, decay constant,
         decay power, number
        '''

        zsiz = len(z)
        # Variables suitable for C++
        dt_32 = np.float32(dt)
        x0i_32 = np.float32(X0i)
        c0_32 = np.float32(c0)
        c1_32 = np.float32(c1)
        alpha_32 = np.float32(alpha)
        a_32 = np.float32(a)
        k_32 = np.float32(k)
        n_32 = np.float32(n)
        s_32 = np.float32(s)
        gx = np.float32(g[0])
        gy = np.float32(g[1])
        gz = np.float32(g[2])
        pix2nm_32 = np.float32(pix2nm)

        mf = cl.mem_flags

        # Set up C++ <-> .py buffers
        buf_size = xsiz * ysiz * 4  # 4 bytes for 32-bit
        shape = np.array([xsiz, ysiz], dtype=np.int32)
        buff_0 = np.zeros((xsiz, ysiz), dtype=np.float32)
        # An array used for adding, initialised with the constant s
        # buff_s = np.ones((xsiz, ysiz), dtype=np.float32) * s
        # for complex amplitudes
        # NB 4 for BF,DF,Re,Im & 4 for 32-bit
        amp_buf_size = xsiz * ysiz * 4 * 4

        # C++ <-> .py buffers

        # for deviation parameter s
        s_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dR_x_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dR_y_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dR_z_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dRz_x_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dRz_y_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dRz_z_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        # for extinction distance Xgr
        Xg_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dP_x_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dP_y_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        dP_z_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        # NB have to use temp buffers because source & target buffers
        # cannot be the same
        t0_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        tx_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        ty_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        tz_array = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)

        # for the image
        # Complex wave amplitudes are held in F = [BF,DF]
        # generate F values at the top [1,0] for all pixels in the buffer
        wave_in = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wave_in_array = np.tile(wave_in, xsiz*ysiz).astype(np.float32)
        wave_out_array = wave_in_array*0.0
        in_out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, amp_buf_size)
        cl.enqueue_copy(self.queue, in_out_buf, wave_in_array)

        # work through the slices
        for i in range(zsiz):
            z_32 = np.float32(z[i])
            dz = np.float32(0.1)
            zdz = np.float32(z_32 + dz)
            # initialise arrays for this slice
            cl.enqueue_copy(self.queue, s_array, buff_0)
            cl.enqueue_copy(self.queue, dP_x_array, buff_0)
            cl.enqueue_copy(self.queue, dP_y_array, buff_0)
            cl.enqueue_copy(self.queue, dP_z_array, buff_0)
            cl.enqueue_copy(self.queue, dR_x_array, buff_0)
            cl.enqueue_copy(self.queue, dR_y_array, buff_0)
            cl.enqueue_copy(self.queue, dR_z_array, buff_0)
            cl.enqueue_copy(self.queue, dRz_x_array, buff_0)
            cl.enqueue_copy(self.queue, dRz_y_array, buff_0)
            cl.enqueue_copy(self.queue, dRz_z_array, buff_0)
            cl.enqueue_copy(self.queue, t0_array, buff_0)
            cl.enqueue_copy(self.queue, tx_array, buff_0)
            cl.enqueue_copy(self.queue, ty_array, buff_0)
            cl.enqueue_copy(self.queue, tz_array, buff_0)
            cl.enqueue_copy(self.queue, Xg_array, buff_0)

            # work through the vortices, there are m of them
            for j in range(m):
                # 1 gives a single y-array, 2 gives x- and y-arrays
                for h in range(2):
                    if (h == 0):
                        c0_32 = np.float32(0.1*c0)
                    else:
                        c0_32 = np.float32(c0)
                    # line direction u is 0 for y and 1 for x
                    u = np.float32(h)
                    # position of current vortex
                    pos_32 = np.float32((j+1)*ysiz/(m+1))
                    # sign of current vortex (alternate +/-)
                    sense = np.float32((-1)**j)

                    # calculate polarisation displacements for Xg
                    self.image_prog.calculate_dR(self.queue, shape, None,
                                                 tx_array, ty_array,
                                                 tz_array, pix2nm_32,
                                                 z_32, c1_32, alpha_32, n_32,
                                                 pos_32, sense, u)
                    # add them onto the total
                    cl.enqueue_copy(self.queue, t0_array, dP_x_array)
                    self.image_prog.update(self.queue, shape, None, dP_x_array,
                                           tx_array, t0_array)
                    cl.enqueue_copy(self.queue, t0_array, dP_y_array)
                    self.image_prog.update(self.queue, shape, None, dP_y_array,
                                           ty_array, t0_array)
                    cl.enqueue_copy(self.queue, t0_array, dP_z_array)
                    self.image_prog.update(self.queue, shape, None, dP_z_array,
                                           tz_array, t0_array)

                    # calculate strain displacements for s at position z
                    self.image_prog.calculate_dR(self.queue, shape, None,
                                                 tx_array, ty_array,
                                                 tz_array, pix2nm_32,
                                                 z_32, c0_32, alpha_32, n_32,
                                                 pos_32, sense, u)
                    # add them onto the total
                    cl.enqueue_copy(self.queue, t0_array, dR_x_array)
                    self.image_prog.update(self.queue, shape, None, dR_x_array,
                                           tx_array, t0_array)
                    cl.enqueue_copy(self.queue, t0_array, dR_y_array)
                    self.image_prog.update(self.queue, shape, None, dR_y_array,
                                           ty_array, t0_array)
                    cl.enqueue_copy(self.queue, t0_array, dR_z_array)
                    self.image_prog.update(self.queue, shape, None, dR_z_array,
                                           tz_array, t0_array)

                    # calculate strain displacements for s at position z+dz
                    self.image_prog.calculate_dR(self.queue, shape, None,
                                                 tx_array, ty_array,
                                                 tz_array, pix2nm_32,
                                                 zdz, c0_32, alpha_32, n_32,
                                                 pos_32, sense, u)
                    # add them onto the total
                    cl.enqueue_copy(self.queue, t0_array, dRz_x_array)
                    self.image_prog.update(self.queue, shape, None,
                                           dRz_x_array,
                                           tx_array, t0_array)
                    cl.enqueue_copy(self.queue, t0_array, dRz_y_array)
                    self.image_prog.update(self.queue, shape, None,
                                           dRz_y_array,
                                           ty_array, t0_array)
                    cl.enqueue_copy(self.queue, t0_array, dRz_z_array)
                    self.image_prog.update(self.queue, shape, None,
                                           dRz_z_array,
                                           tz_array, t0_array)

            # all displacements have been calculated
            # now calculate s and Xg
            self.image_prog.calculate_Xg(self.queue, shape, None, Xg_array,
                                         dP_x_array, dP_y_array, dP_z_array,
                                         gx, gy, gz, a_32, k_32)
            self.image_prog.calculate_s(self.queue, shape, None, s_array,
                                        dR_x_array, dR_y_array, dR_z_array,
                                        dRz_x_array, dRz_y_array, dRz_z_array,
                                        gx, gy, gz, a_32, dz, s_32)

            # Howie-Whelan wave prpopagation
            self.image_prog.propagate_wave(self.queue, shape, None, in_out_buf,
                                           dt_32, s_array, Xg_array, x0i_32)
            cl.enqueue_copy(self.queue, wave_out_array, in_out_buf)
            cl.enqueue_copy(self.queue, in_out_buf, wave_out_array)

        # wave propagation is complete
        # Convert to intensity
        self.image_prog.amplitude2intensity(self.queue, shape, None,
                                            in_out_buf)

        output = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, output, in_out_buf)

        # debug
        # dbg_out = np.zeros((shape[0], shape[1]), dtype=np.float32)
        # cl.enqueue_copy(self.queue, dbg_out, s_array)
        # image_bf = dbg_out

        image_bf = np.flip(output[:, :, 0], axis=0)
        image_df = np.flip(output[:, :, 2], axis=0)

        return image_bf, image_df
